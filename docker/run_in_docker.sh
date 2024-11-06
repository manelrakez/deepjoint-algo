#!/bin/bash

set -e

function usage() {
  local progname=$(basename "$0")
  echo "Usage:"
  echo "  ${progname} --image|i <docker-image-name> [--container_name|-c <container name>] <command> [args ...]"
  echo ""
  echo "You can set an env variable 'BASHRC_DOCKER', which is a path to an hand-defined bashrc"
  echo "with export/alias statements, his content will be added to .bashrc in the container"
  echo ""
  echo ""
  echo "Examples:"
  echo "  ${progname} bash"
  echo "  ${progname} jupyter <port>"
  echo "  ${progname} tensorboard <port> <logdir>"
  exit 1
}


function get_switch_user_cmd() {
  local uid="$(id -u)"
  local gid=$(id -g)
  local username=$(id -n -u)
  local groupname=$(id -n -g)

  local cmdline="rm /bin/sh && ln -s /bin/bash /bin/sh "
  cmdline+="; groupadd -f ${groupname} && groupmod -o -g ${gid} ${groupname}"
  cmdline+="; id -u ${username} &>/dev/null || useradd -N ${username} && usermod -o -u ${uid} -g ${gid} ${username}"
  cmdline+="; mkdir /user_home && chown -R ${username}:${username} /user_home/ "
  cmdline+="; chroot --userspec=${username} --skip-chdir / "

  echo "${cmdline}"
}


function get_random_word() {
  # pick random word, delete all but alphanumerical characters, convert to lower case
  random_word=$(shuf -n1 /usr/share/dict/words | tr -dc "[:alnum:]" | tr "[:upper:]" "[:lower:]")
  echo "${random_word}"
}


function get_random_container_name() {
  short_username="${USER:0:5}"
  random_word=$(get_random_word)
  random_word_bis=$(get_random_word)
  name="${short_username}_${random_word}_${random_word_bis}"
  echo "${name}"
}

(( $# < 1 )) && usage

# default image name and container_name
IMAGE="" # must be updated after parsing input args
CONTAINER_NAME="$(get_random_container_name)"


while [[ $# -gt 1 ]]; do
  case "$1" in
    -c|--container_name)
      if [ ! -z "$2" ]; then
        echo "--container_name provided and not empty : $2"
        CONTAINER_NAME="$2"
      else
        echo "--container_name provided, but empty. Keep : $CONTAINER_NAME"
      fi
      shift 2;;
    -i|--image)
      IMAGE="$2"; shift 2;;
    --gpus)
      RUN_OPTS+=(--gpus all); shift 2;;
    *)
      break;;
  esac
done

if [ ${IMAGE} == "" ]; then
    echo "--image/-i was not provided. Required."
    exit 1
fi


GENERATED_BASHRC_PATH="${PWD}/${CONTAINER_NAME}.bashrc"
touch "${GENERATED_BASHRC_PATH}"

{ # https://github.com/koalaman/shellcheck/wiki/SC2129
  echo "export PS1=\"(\H) \[\e[31m\]in-docker\[\e[m\] \[\e[33m\]\w\[\e[m\] > \" "
  echo "export TERM=xterm-256color"
  echo "alias grep='grep --color=auto'"
  echo "alias ls='ls --color=auto'"
  echo "alias ll='ls -lh'"
} >> "${GENERATED_BASHRC_PATH}"

HOST_HOME="$HOME"

# Set docker run options
RUN_OPTS=(-it --rm --network=host --hostname="$(uname -n)" --workdir="${PWD}")

# This is a sane default to use, as tensorflow can be greedy with gpus
RUN_OPTS+=(-e CUDA_VISIBLE_DEVICES="")
RUN_OPTS+=(--name="$CONTAINER_NAME")
RUN_OPTS+=(-e HOME=/user_home -e USER="${USER}")
RUN_OPTS+=(-e GENERATED_BASHRC_PATH="${GENERATED_BASHRC_PATH}" -e BASH_ENV=${GENERATED_BASHRC_PATH} )

# ------------ ADD VOLUMES ----------------
## add 'PWD' from host
RUN_OPTS+=(--mount type=bind,source=${PWD},target=${PWD})

## add local homedir  only if not $PWD
if [[ "${PWD}" != "${HOST_HOME}" ]];then
    RUN_OPTS+=(--mount type=bind,source=${HOST_HOME},target=${HOST_HOME})
fi
# ------------ ADD VOLUMES (end) ----------------

# identify action to perform in docker : bash ? python ? jupyter ?
if [[ "$1" == "bash" ]];then
    CMD="bash --rcfile ${GENERATED_BASHRC_PATH} "
elif [[ "$1" == "jupyter" ]];then
    CMD="sh -c 'source ${GENERATED_BASHRC_PATH}; jupyter notebook --ip=0.0.0.0 --no-browser --port=$2'"
elif [[ "$1" == "tensorboard" ]];then
    CMD="sh -c 'source ${GENERATED_BASHRC_PATH}; CUDA_VISIBLE_DEVICES= tensorboard --host=0.0.0.0 --port=$2 --logdir=$3'"
else
    usage
fi


# non-empty CMD ?
[[ "${CMD}" = "" ]] && usage
[[ ! -x $(command -v docker) ]] && echo "ERROR: 'docker' command missing from PATH." && usage

##  to un-comment for debug :
#echo -e "\nimage: \n ${IMAGE} \n"
#echo -e "\ncmd: \n ${CMD} \n"
#echo -e "\nrun opts: \n ${RUN_OPTS[@]} \n"
#echo -e "\nswitch_user: \n $(get_switch_user_cmd) \n"

docker run "${RUN_OPTS[@]}" "${IMAGE}" bash -c "$(get_switch_user_cmd) ${CMD}"

rm "${GENERATED_BASHRC_PATH}"
