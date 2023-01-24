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
  ];

  # buildPhase = "c++ -o main main.cpp -lPocoFoundation -lboost_system";

  # installPhase = ''
  #  mkdir -p $out/bin
  #  cp main $out/bin/
  # '';
}
