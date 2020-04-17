let 
  pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    doCheck = false;
#    buildInputs = with pkgs; [ _python37 ];

    buildInputs = with pkgs; [ (pkgs.python37.withPackages (ps: with ps; [ pip numpy pandas cython scikitlearn pyprof2calltree memory_profiler deap ])) ];
    
    shellHook = ''
            alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' \pip"
            export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.7/site-packages:$PYTHONPATH"
            unset SOURCE_DATE_EPOCH
    '';
  }
