# SPDX-FileCopyrightText: 2024 Joshua Jewell
# SPDX-License-Identifier: MIT
{
  description = "VAE Dataset Normalizer - Normalize VAE-decoded image datasets with formal verification";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    naersk = {
      url = "github:nix-community/naersk";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, naersk }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
        };

        naersk-lib = pkgs.callPackage naersk {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };

      in {
        packages = {
          default = naersk-lib.buildPackage {
            src = ./.;
            pname = "vae-normalizer";
            version = "1.0.0";

            nativeBuildInputs = with pkgs; [
              pkg-config
            ];

            buildInputs = with pkgs; [
              openssl
            ];

            meta = with pkgs.lib; {
              description = "Normalize VAE-decoded image datasets with formal verification";
              homepage = "https://huggingface.co/datasets/joshuajewell/VAEDecodedImages-SDXL";
              license = licenses.mit;
              maintainers = [ ];
              platforms = platforms.unix;
            };
          };

          container = pkgs.dockerTools.buildImage {
            name = "vae-normalizer";
            tag = "latest";
            copyToRoot = pkgs.buildEnv {
              name = "vae-normalizer-root";
              paths = [ self.packages.${system}.default ];
              pathsToLink = [ "/bin" ];
            };
            config = {
              Cmd = [ "/bin/vae-normalizer" ];
              WorkingDir = "/data";
              Volumes = { "/data" = {}; };
            };
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust
            rustToolchain
            pkg-config
            openssl

            # Build tools
            just
            gnumake

            # Linting & formatting
            clippy
            rustfmt

            # Optional: verification tools
            cue
            nickel

            # Documentation
            mdbook
          ];

          shellHook = ''
            echo "VAE Dataset Normalizer development environment"
            echo "Rust: $(rustc --version)"
            echo ""
            echo "Commands:"
            echo "  just build    - Build release binary"
            echo "  just test     - Run tests"
            echo "  just validate - Run RSR compliance checks"
          '';

          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
        };

        # RSR compliance checks
        checks = {
          format = pkgs.runCommand "check-format" {
            buildInputs = [ rustToolchain ];
          } ''
            cd ${./.}
            cargo fmt --check
            touch $out
          '';

          clippy = pkgs.runCommand "check-clippy" {
            buildInputs = [ rustToolchain pkgs.clippy ];
          } ''
            cd ${./.}
            cargo clippy -- -D warnings
            touch $out
          '';

          tests = pkgs.runCommand "check-tests" {
            buildInputs = [ rustToolchain ];
          } ''
            cd ${./.}
            cargo test
            touch $out
          '';
        };
      }
    );
}
