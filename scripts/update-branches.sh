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

# Check if develop is behind main
behind_count=$(git rev-list --left-right --count develop...main | cut -f2)

if [ "$behind_count" -gt 0 ]; then
  git rebase main
  git push
fi

# Switch back to the original branch
git checkout "$current_branch"

# If the current branch is not main or develop, rebase it onto develop
if [[ "$current_branch" != "main" && "$current_branch" != "develop" ]]; then
  git rebase develop
fi

echo "Branches have been updated and rebased accordingly."
