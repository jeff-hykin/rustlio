use crate::commons::*;
use crate::ieskf::{IESKF, State};
use crate::imu_processor::IMUProcessor;
use crate::lidar_processor::LidarProcessor;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BuilderStatus {
    ImuInit,
    MapInit,
    Mapping,
}

pub struct MapBuilder {
    config: Config,
    status: BuilderStatus,
    pub kf: IESKF,
    imu_processor: IMUProcessor,
    pub lidar_processor: LidarProcessor,
    last_good_state: Option<State>,
}

impl MapBuilder {
    pub fn new(config: Config) -> Self {
        let mut kf = IESKF::new();
        kf.set_max_iter(config.ieskf_max_iter);

        let imu_processor = IMUProcessor::new(&config);
        let lidar_processor = LidarProcessor::new(&config);

        MapBuilder {
            config,
            status: BuilderStatus::ImuInit,
            kf,
            imu_processor,
            lidar_processor,
            last_good_state: None,
        }
    }

    pub fn status(&self) -> BuilderStatus {
        self.status
    }

    pub fn process(&mut self, package: &mut SyncPackage) {
        if self.status == BuilderStatus::ImuInit {
            if self.imu_processor.initialize(package, &mut self.kf) {
                self.status = BuilderStatus::MapInit;
            }
            return;
        }

        self.imu_processor.undistort(package, &mut self.kf);

        if self.status == BuilderStatus::MapInit {
            let r_wl = self.lidar_processor.r_wl(&self.kf);
            let t_wl = self.lidar_processor.t_wl(&self.kf);
            let cloud_world =
                LidarProcessor::transform_cloud(&package.cloud, &r_wl, &t_wl);
            self.lidar_processor.init_cloud_map(&cloud_world);
            self.status = BuilderStatus::Mapping;
            return;
        }

        let state_pre_update = self.kf.x.clone();
        self.lidar_processor.update(package, &mut self.kf);

        if !self.reject_if_over_speed(state_pre_update) {
            self.lidar_processor.incr_cloud_map(&self.kf);
        }
    }

    /// Velocity-cap guardrail. A single-frame IESKF blow-up shows up as an
    /// implausible post-update velocity. When `kf.x.v` exceeds `max_velocity`,
    /// roll the state back to the last good one (or the pre-update state if we
    /// have none yet), zero its velocity, and report rejection so the caller
    /// skips the map insert — keeping the bad pose out of the ikd-tree. State is
    /// held per-instance, so independent `MapBuilder`s never interfere.
    /// Returns `true` when the frame was rejected.
    fn reject_if_over_speed(&mut self, state_pre_update: State) -> bool {
        let post_update_speed = self.kf.x.v.norm();
        if self.config.max_velocity > 0.0 && post_update_speed > self.config.max_velocity {
            let mut recovered_state =
                self.last_good_state.clone().unwrap_or(state_pre_update);
            recovered_state.v = V3D::zeros();
            self.kf.x = recovered_state;
            true
        } else {
            self.last_good_state = Some(self.kf.x.clone());
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_cap_rolls_back_and_stays_instance_local() {
        let mut config = Config::default();
        config.max_velocity = 3.0;

        let mut mapper = MapBuilder::new(config.clone());
        let second_mapper = MapBuilder::new(config);

        // A sane frame is accepted and remembered as the last good state.
        mapper.kf.x.t_wi = V3D::new(1.0, 0.0, 0.0);
        mapper.kf.x.v = V3D::new(1.0, 0.0, 0.0);
        let pre_update_state = mapper.kf.x.clone();
        assert!(!mapper.reject_if_over_speed(pre_update_state));
        assert!(mapper.last_good_state.is_some());

        // An over-speed frame is rejected: velocity zeroed, pose rolled back to
        // the last good one, not the teleported pre-update value.
        let pre_update_state = mapper.kf.x.clone();
        mapper.kf.x.t_wi = V3D::new(99.0, 0.0, 0.0);
        mapper.kf.x.v = V3D::new(50.0, 0.0, 0.0);
        assert!(mapper.reject_if_over_speed(pre_update_state));
        assert_eq!(mapper.kf.x.v, V3D::zeros());
        assert_eq!(mapper.kf.x.t_wi, V3D::new(1.0, 0.0, 0.0));

        // The second mapper shares no state with the first.
        assert!(second_mapper.last_good_state.is_none());
    }
}
