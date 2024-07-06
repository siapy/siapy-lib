#!/usr/bin/env bash

set -e
set -x

# Save the current branch name
current_branch=$(git branch --show-current)

# Update main
git checkout main
git pull origin main

# Update develop
git checkout develop
git pull origin develop
git rebase main
git push

# Switch back to the original branch
git checkout "$current_branch"

# If the current branch is not main or develop, rebase it onto develop
if [[ "$current_branch" != "main" && "$current_branch" != "develop" ]]; then
  git rebase develop
fi

echo "Branches have been updated and rebased accordingly."
