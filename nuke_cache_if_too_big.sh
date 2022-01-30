function nuke_if_too_big() {
  path=$1
  limit_mb=$2
  size_mb=$(du -m -d0 ${path} | cut -f 1)
  if (( ${size_mb} > ${limit_mb} )); then
    echo "${path} is too large (${size_mb}mb), nuking it."
    rm -rf ${path}
  fi
}

nuke_if_too_big ~/.cache/pants/lmdb_store 2048
nuke_if_too_big ~/.cache/pants/setup 768
nuke_if_too_big ~/.cache/pants/named_caches 2048

echo "Contents of setup cache:"
ls -l /home/runner/.cache/pants/setup/
echo "Contents of linux cache"
ls -l /home/runner/.cache/pants/setup/bootstrap-Linux-x86_64
echo "contents of python cache"
ls -l /home/runner/.cache/pants/setup/bootstrap-Linux-x86_64/pants.eTLSta/install
echo "contents of python cache bin"
ls -l /home/runner/.cache/pants/setup/bootstrap-Linux-x86_64/pants.eTLSta/install/bin
echo "python versiom"
/opt/hostedtoolcache/Python/3.9.9/x64/bin/python3.9 --version
echo "python version through symlink"
/home/runner/.cache/pants/setup/bootstrap-Linux-x86_64/2.8.0_py39/bin/python --version
