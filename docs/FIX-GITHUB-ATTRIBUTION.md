# Removing Claude from the repo's authorship and Contributors

Run this **on your Mac**, in one go. It takes about 30 seconds.

## Why it has to be you

The rewrite cannot be pushed from the Claude Code container — force-pushing a default branch
is blocked there by a safety check, and correctly so. The commands below do exactly the same
thing locally.

**Run this last**, once the session has pushed its final work, so the rewrite covers every
commit rather than needing a second pass.

## The problem, precisely

Two separate things make Claude appear:

1. **Contributors list.** GitHub builds it from the **default branch only**. `main` carries
   `Co-Authored-By: Claude Sonnet 4.6` in commit `7405109`, so Claude is listed regardless of
   what any other branch says.
2. **`theakbarjuraev` vs `abyyworld`.** GitHub ignores the `name` field in a commit and
   resolves the **email** to an account. Your repo has two:

   | Email | GitHub shows |
   |---|---|
   | `annolieberto@gmail.com` | **abyyworld** |
   | `kenny09077@gmail.com` | **theakbarjuraev** |

   Some commits were made under the second one. Rewriting everything to
   `annolieberto@gmail.com` makes it all read `abyyworld`.

## The fix

```bash
# Fresh clone so nothing local interferes
cd ~/Desktop
rm -rf attribution-fix && git clone https://github.com/abyyworld/Drone-visualisation-training.git attribution-fix
cd attribution-fix

# Make sure every branch is present locally, so --all actually covers them
git fetch origin '+refs/heads/*:refs/heads/*' 2>/dev/null || true
git branch -a

# Rewrite: one identity everywhere, no Claude trailers anywhere
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
  --env-filter 'export GIT_AUTHOR_NAME="abyyworld"
                export GIT_AUTHOR_EMAIL="annolieberto@gmail.com"
                export GIT_COMMITTER_NAME="abyyworld"
                export GIT_COMMITTER_EMAIL="annolieberto@gmail.com"' \
  --msg-filter 'grep -viE "^(Co-Authored-By: Claude|Claude-Session: )" | cat -s || true' \
  -- --all
```

### Check before you push

```bash
# Should print exactly one line: abyyworld <annolieberto@gmail.com>
git log --all --format="%an <%ae>" | sort -u

# Should print 0
git log --all --format="%B" | grep -ic "claude"
```

If either looks wrong, stop — nothing has been pushed yet, so just delete the folder.

### Push

```bash
git push --force origin main
git push --force origin claude/model-retrain-solar-panels-n77z3v
```

### Afterwards

```bash
cd ~/Desktop && rm -rf attribution-fix
```

Any **existing clone you have elsewhere is now broken** — its history no longer matches. Delete
it and re-clone, or `git fetch origin && git reset --hard origin/main`.

The Contributors list can take a few minutes to a few hours to refresh; GitHub recomputes it
on a schedule rather than instantly. The commit list updates immediately.

## What this does not change

Commits will show as **Unverified**. That badge only appears when the platform signs commits
under its own identity — once they are authored as you, this environment has no key for your
identity. It is cosmetic: it does not affect merging, CI, or anything functional.

To get both your name *and* a Verified badge, set up commit signing on your Mac:

```bash
ssh-keygen -t ed25519 -C "annolieberto@gmail.com" -f ~/.ssh/git-signing
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/git-signing.pub
git config --global commit.gpgsign true
```

Then add `~/.ssh/git-signing.pub` to GitHub under **Settings → SSH and GPG keys → New SSH key
→ Key type: Signing Key**. Commits you make from your Mac after that will be Verified. The
historical ones stay unverified unless you re-sign them, which is rarely worth it.
