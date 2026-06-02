{
  description = "Standalone C++ relocalization benchmark harnesses (reloc_bench + global_reloc_bench)";

  # Relocalizer uses only PCL + Eigen (no GTSAM), so plain nixpkgs is enough —
  # no need to ABI-match an external GTSAM build the way loop_closure_bench does.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        deps = [ pkgs.eigen pkgs.boost pkgs.tbb pkgs.flann pkgs.pcl ];
        # The relocalizer source lives in ../../localizer; pull it in for the build.
        build = pkgs.stdenv.mkDerivation {
          pname = "reloc_bench";
          version = "0.1.0";
          src = ../..;
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = deps;
          # Build only the harness subdir (it references ../../localizer via CMake).
          configurePhase = "cmake -S reloc_bench/harness -B build -DCMAKE_BUILD_TYPE=Release";
          buildPhase = "cmake --build build -j $NIX_BUILD_CORES";
          installPhase = ''
            mkdir -p $out/bin
            cp build/reloc_bench build/global_reloc_bench $out/bin/
          '';
        };
      in {
        packages.default = build;
        packages.reloc_bench = build;
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = deps;
        };
      });
}
