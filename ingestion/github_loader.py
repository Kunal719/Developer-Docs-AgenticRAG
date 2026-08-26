from pathlib import Path
from enum import Enum

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

class RepositoryStatus(Enum):
    """
    Status of repository after syncing the local repo from the remote repo
    """
    CLONED = "cloned"
    UPDATED = "updated"
    UNCHANGED = "unchanged"

def _normalize(paths):
    return {p.strip("/") for p in paths}

def load_repository(repo_url: str, sparse_paths: list[str], local_path: Path, branch: str = "main", check_for_updates: bool = True,) -> tuple[Path, RepositoryStatus]:
    """
    Clone or update a GitHub repository using sparse checkout.

    Only the required documentation directory is checked out to reduce
    download size and startup time.

    The local repository is treated as a read-only mirror of the upstream
    documentation. Updates are synchronized by resetting to the latest remote
    commit instead of merging local history.

    Args:
        repo_url: GitHub repository URL.
        local_path: Local repository directory.
        branch: Branch to checkout.
        check_for_updates: Pull latest changes if repository already exists.

    Returns:
        A tuple containing the local repository path and the repository status.
    """

    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if (local_path / ".git").exists():
            repo = Repo(local_path)

            if not check_for_updates:
                return local_path, RepositoryStatus.UNCHANGED

            print("Checking for repository updates...")

            repo.remotes.origin.fetch(
                depth=1,
                filter="blob:none",
            )

            local_commit = repo.head.commit
            remote_commit = repo.refs[f"origin/{branch}"].commit

            # TODO:
            # Instead of comparing repository commits, compare only the tracked
            # documentation directory against origin/<branch>. This would avoid
            # pulling the repository when unrelated files outside the sparse
            # checkout path have changed.

            try:
                current_paths = _normalize(repo.git.sparse_checkout("list").splitlines())
            except GitCommandError:
                current_paths = set()

            paths_changed = current_paths != _normalize(sparse_paths)

            if local_commit.hexsha != remote_commit.hexsha:
                # This repository is treated as a read-only mirror of the upstream
                # documentation. Rather than merging histories with `git pull`, we
                # simply make the local checkout identical to origin/<branch>.
                repo.git.reset("--hard", f"origin/{branch}")
                status = RepositoryStatus.UPDATED
            elif paths_changed:
                status = RepositoryStatus.UPDATED
            else:
                status = RepositoryStatus.UNCHANGED

            repo.git.sparse_checkout("set", *sparse_paths)

            if status == RepositoryStatus.UPDATED:
                print("Repository updated successfully.")
            else:
                print("Repository is already up to date.")

            return local_path, status

        print("Cloning documentation repository...")

        repo = Repo.clone_from(
            repo_url,
            local_path,
            branch=branch,
            depth=1,
            filter="blob:none",
            no_checkout=True,
        )

        repo.git.sparse_checkout("init", "--no-cone")
        repo.git.sparse_checkout("set", *sparse_paths)

        repo.git.checkout(branch)

        print("Repository cloned successfully.")

        return local_path, RepositoryStatus.CLONED

    except (GitCommandError, InvalidGitRepositoryError) as exc:
        raise RuntimeError(
            f"Failed to prepare repository: {exc}"
        ) from exc