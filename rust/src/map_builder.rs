use crate::commons::*;
use crate::ieskf::IESKF;
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

        self.lidar_processor.process(package, &mut self.kf);
    }
}
