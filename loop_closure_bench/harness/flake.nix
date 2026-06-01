{
  description = "Standalone C++ PGO loop-closure benchmark harness (pgo_bench)";

  inputs = {
    # Follow gtsam-extended's nixpkgs so PCL / Eigen / Boost are the SAME
    # versions GTSAM was compiled against -- mixing Eigen majors across the
    # PCL<->GTSAM ABI boundary is what broke the brew build (Eigen 5.0.1).
    gtsam-extended.url = "github:jeff-hykin/gtsam-extended";
    nixpkgs.follows = "gtsam-extended/nixpkgs";
    flake-utils.follows = "gtsam-extended/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, gtsam-extended }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        gtsam = gtsam-extended.packages.${system}.gtsam-cpp;
        deps = [
          pkgs.eigen
          pkgs.boost
          pkgs.tbb
          pkgs.flann
          pkgs.pcl
          gtsam
        ];
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = deps;
        };
      });
}
