{
  description = "pgo_bench_rs — Rust port of the point-to-plane PGO loop closure";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        rustToolchain = pkgs.rust-bin.stable.latest.default;
      in {
        # `nix run .#pgo_bench_rs -- --clouds ... ` always builds fresh from
        # source (no stale-binary trap), since the derivation rebuilds on change.
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "pgo_bench_rs";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [ rustToolchain pkgs.pkg-config ];
        };
      });
}
