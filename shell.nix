with import <nixpkgs> {};
{ pkgs ? import <nixpkgs> {} }:

stdenv.mkDerivation {
  name = "nnalgorithm";
  src = ./.;

  buildInputs = [ 
  	git 
	gtest 
	gbenchmark 
	gnumake 
	cmake 
	clang 
	ripgrep
	ninja
	linuxKernel.packages.linux_latest_libre.perf
	flamegraph
  ];

  # buildPhase = "c++ -o main main.cpp -lPocoFoundation -lboost_system";
  LIBCLANG_PATH = "${pkgs.llvmPackages_14.libclang.lib}/lib";
  # installPhase = ''
  #  mkdir -p $out/bin
  #  cp main $out/bin/
  # '';
}
