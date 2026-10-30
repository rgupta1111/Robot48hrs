"""Generic service that makes an arm follow a named object (e.g. a cup) seen by a camera + vision service.

Runs on the machine next to viam-server. Starts DISENGAGED: nothing moves until DoCommand {"command": "engage"}.
"""
import asyncio
import time
from typing import ClassVar, Mapping, Sequence

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.vision import VisionClient
from viam.utils import ValueTypes, struct_to_dict

from follow_logic import Detection, FollowLogic, FollowSettings
from mimic.follower import FollowConfig
from tidy.robot import TableToWorld

LOGGER = getLogger(__name__)
COMMANDS = "engage, disengage, stop, status, recenter, set_lock"


def _floats(value, n, default):
    value = default if value is None else value
    if len(value) != n:
        raise ValueError(f"expected {n} numbers, got {list(value)}")
    return tuple(float(v) for v in value)


class ObjectFollowService(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("<ns>", "robot48hrs"), "object-follower")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        svc = super().new(config, dependencies)
        svc.reconfigure(config, dependencies)
        return svc

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        attrs = struct_to_dict(config.attributes)
        missing = [k for k in ("arm", "camera", "vision") if not attrs.get(k)]
        if missing:
            raise ValueError(f"missing required attribute(s): {', '.join(missing)}")
        _floats(attrs.get("bounds"), 6, (0,) * 6)
        _floats(attrs.get("table_to_world"), 4, (0,) * 4)
        return [attrs["arm"], attrs["camera"], attrs["vision"]]

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self._shutdown()
        a = struct_to_dict(config.attributes)
        self._arm = dependencies[Arm.get_resource_name(a["arm"])]
        self._camera_name = a["camera"]
        self._vision = dependencies[VisionClient.get_resource_name(a["vision"])]
        self._table = TableToWorld(*_floats(a.get("table_to_world"), 4, (0, 0, 0, 0)))
        cfg = FollowConfig(
            bounds=_floats(a.get("bounds"), 6, FollowConfig().bounds),
            deadband_mm=float(a.get("deadband_mm", 10.0)), gain=float(a.get("gain", 0.5)),
            max_step_mm=float(a.get("max_step_mm", 60.0)), max_speed_mm_s=float(a.get("max_speed_mm_s", 80.0)),
            smoothing=float(a.get("smoothing", 0.5)), lost_disengage_s=float(a.get("lost_disengage_s", 3.0)))
        self._logic = FollowLogic(FollowSettings(
            lock=str(a.get("lock", "cup")), min_confidence=float(a.get("min_confidence", 0.3)),
            mm_per_px=float(a.get("mm_per_px", 0.8)), swap_xy=bool(a.get("swap_xy", False)),
            flip_u=bool(a.get("flip_u", False)), flip_v=bool(a.get("flip_v", False)), follow=cfg))
        self._period = 1.0 / max(float(a.get("loop_hz", 4.0)), 0.1)
        self._orient = None
        self._error = None
        self._ticks = 0
        self._stop = asyncio.Event()
        self._task = asyncio.ensure_future(self._loop())

    # --- control loop -------------------------------------------------------------
    async def _loop(self):
        t0 = time.monotonic()
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                if self._logic.engaged:
                    await self._tick(started - t0)
                else:
                    self._logic.status = "idle"
            except asyncio.CancelledError:
                raise
            except Exception as e:  # fail safe: stop following and report it via status
                self._error = f"{type(e).__name__}: {e}"
                LOGGER.error("follow loop error, disengaging: %s", self._error)
                self._logic.disengage()
                await self._safe_stop()
            await asyncio.sleep(max(self._period - (time.monotonic() - started), 0.02))

    async def _tick(self, t):
        res = await self._vision.get_detections_from_camera(self._camera_name)
        dets = [Detection(d.x_min, d.y_min, d.x_max, d.y_max, d.confidence, d.class_name) for d in res]
        pose = await self._arm.get_end_position()
        if self._orient is None:
            self._orient = pose           # hold the orientation from engage so the camera keeps looking the same way
        cur = self._table.to_robot(pose.x, pose.y, pose.z)
        new = self._logic.step(dets, cur, t)
        self._ticks += 1
        if new is not None:
            wx, wy, wz = self._table.apply(*new)
            o = self._orient
            await self._arm.move_to_position(Pose(x=wx, y=wy, z=wz, o_x=o.o_x, o_y=o.o_y, o_z=o.o_z, theta=o.theta))

    async def _safe_stop(self):
        try:
            await self._arm.stop()
        except Exception as e:
            LOGGER.error("arm.stop failed: %s", e)

    # --- API ------------------------------------------------------------------------
    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout=None, **kwargs) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        lg = self._logic
        if cmd == "engage":
            self._error, self._orient = None, None
            lg.engage(time.monotonic())
        elif cmd == "disengage":
            lg.disengage()
        elif cmd == "stop":
            lg.disengage()
            await self._safe_stop()
        elif cmd == "recenter":      # re-zero: current object position = current gripper position
            lg._ref = None
        elif cmd == "set_lock":
            if not command.get("lock"):
                raise ValueError('set_lock needs "lock": "<class name>"')
            lg.set_lock(str(command["lock"]))
        elif cmd != "status":
            raise ValueError(f"unknown command {cmd!r}; use one of: {COMMANDS}")
        return {"engaged": lg.engaged, "status": lg.status, "lock": lg.s.lock, "object_px": list(lg.object_px or []),
                "target_mm": [round(v, 1) for v in (lg.target or [])], "ticks": self._ticks, "error": self._error or ""}

    # --- lifecycle -------------------------------------------------------------------
    def _shutdown(self):
        if getattr(self, "_task", None) is not None:
            self._stop.set()
            self._task.cancel()
            self._task = None

    async def close(self):
        self._logic.disengage()
        self._shutdown()
        await self._safe_stop()
