from packaging.version import Version
from setuptools_scm import get_version

UPSTREAM_TAG_DESCRIBE = [
    "git",
    "describe",
    "--dirty",
    "--tags",
    "--long",
    "--match",
    "v[0-9]*",
]


def main() -> None:
    version = get_version(git_describe_command=UPSTREAM_TAG_DESCRIBE)
    Version(version)
    print(version)


if __name__ == "__main__":
    main()
