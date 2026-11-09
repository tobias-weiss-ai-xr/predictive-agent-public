{
  description = "predictive-agent — Nix container image (POMDP decision layer)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      pav = import ./nix/predictive-agent.nix { inherit pkgs; };
    in
    {
      packages.${system} = {
        predictive-agent = pav;
        default = pav;
      };

      # `nix develop` for iterating on the package / building locally.
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [ pkgs.python3 pkgs.python3Packages.psutil pkgs.docker-client ];
      };
    };
}
