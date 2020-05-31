#!/bin/bash -e


pushd() {
    command pushd "$@" > /dev/null
}

popd() {
    command popd "$@" > /dev/null
}


show_help() {
    script_name=$( basename ${BASH_SOURCE[0]} )
    echo "Usage: ${script_name} [OPTION]..."
    cat <<EOD
-h | --help         show help
--lib[=PATH]        path to Tsetlini library code to copy from
EOD
}


function ask_yes_no() {
    while true; do
        read -p "${1}" yn
        case $yn in
            [Yy]* ) echo "Y"; break;;
            [Nn]* ) echo "N"; break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}


function git_root() {
    pushd "${1}"
    local rv=$( git rev-parse --show-toplevel )
    popd
    echo "${rv}"
}


run_work() {
    script_dir=$( dirname ${BASH_SOURCE[0]} )
    script_repo=$( git_root "${script_dir}" )
    echo "PyTsetlini repository path: ${script_repo}"

    lib_repo=$( git_root "${1}" )
    echo "Tsetlin library repository path: ${lib_repo}"

    echo "\$> git status ${script_repo}/libtsetlini"
    pushd "${script_repo}"/libtsetlini
    git status "${script_repo}/libtsetlini"
    popd
    YN=$( ask_yes_no "Is git status OK? " )
    if [[ ${YN} == "N" ]]; then
        return
    fi

    echo "\$> git status ${lib_repo}"
    pushd ${lib_repo}
    lib_commit_hash=$( git rev-parse HEAD )
    lib_commit=$( git log -1 )
    git status "${lib_repo}"
    popd
    YN=$( ask_yes_no "Is git status OK? " )
    if [[ ${YN} == "N" ]]; then
        return
    fi

    echo "Renaming ${script_repo}/libtsetlini"
    mv "${script_repo}"/libtsetlini "${script_repo}"/libtsetlini_backup
    #mkdir "${script_repo}"/libtsetlini

    rsync -av "${lib_repo}"/ "${script_repo}"/libtsetlini

    rm -rfv "${script_repo}"/libtsetlini/bindings
    rm -rfv "${script_repo}"/libtsetlini/.build
    rm -rfv "${script_repo}"/libtsetlini/.git

    echo "${lib_commit_hash}" > "${script_repo}"/libtsetlini/.commit_hash
    echo "${lib_commit}" > "${script_repo}"/libtsetlini/.commit
}


run_main() {
    for i in "$@"
    do
    case ${i} in
        --lib=*)
        ARG_LIB=YES
        LIB_PATH=${i#*=}
        shift
        ;;
        -h|--help)
        ARG_HELP=YES
        shift
        ;;
        *)
        ;;
    esac
    done

    if [[ ${ARG_HELP} == "YES" ]]; then
        show_help
        return
    fi

    if [[ ${ARG_LIB} == "YES" ]]; then
        run_work ${LIB_PATH}
        return
    fi

}


run_main "$@"
exit $?
