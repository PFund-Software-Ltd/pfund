import os

from git import Repo, InvalidGitRepositoryError, NoSuchPathError, Commit


class GitController:
    def __init__(self, file_path: str):
        # find the root directory of the Git repository that contains the given file
        try:
            self._repo = Repo(file_path, search_parent_directories=True)
            self._repo_path = self._repo.git.rev_parse("--show-toplevel")
        except (InvalidGitRepositoryError, NoSuchPathError):
            self._repo = self._repo_path = None
            print(f'{file_path} is not a git repository')

    def is_git_repo(self):
        return self._repo is not None
    
    def get_last_n_commit(self, n=1) -> list[str]:
        if not self._repo:
            raise ValueError('No git repository found')        
        commits: list[Commit] = list(self._repo.iter_commits(paths=self._repo_path, max_count=n))
        return [commit.hexsha for commit in commits]
    
    def commit(self, file_path: str, commit_message: str) -> str | None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'{file_path} not found')
        if not self._repo:
            raise ValueError('No git repository found')
        try:
            file_relpath = os.path.relpath(file_path, self._repo_path)
            changed_files = [d.a_path for d in self._repo.index.diff(None)]
            if file_relpath in self._repo.untracked_files or file_relpath in changed_files:
                self._repo.index.add([file_relpath])  # Stage the file
                commit = self._repo.index.commit(commit_message)  # Commit the changes
                return commit.hexsha  # Return the commit hash
            else:
                print(f"No changes detected in {file_path}.")
                return None
        except Exception as e:
            print(f"Error committing {file_path}: {e}")
            return None
    
    # e.g. git checkout <commit_hash> -- strategy._file_path
    # and git checkout HEAD -- strategy._file_path
    def checkout_file_from_commit(self, commit_hash, file_path):
        """Check out a specific file from a specific commit into the working directory."""
        try:
            commit = self._repo.commit(commit_hash)  # Get the specific commit
            file_relpath = os.path.relpath(file_path, self._repo_path)
            # Access the blob (file content) for the file at the specified commit
            blob = commit.tree / file_relpath
            # Write the content of the blob to the file in the working directory
            with open(file_path, 'wb') as file:
                blob.stream_data(file)
            print(f"File {file_path} has been successfully checked out from commit {commit_hash}.")
        except KeyError:
            print(f"File {file_path} not found in commit {commit_hash}.")
        except Exception as e:
            print(f"Failed to checkout commit {commit_hash}: {e}")
