# wisupaa-whisper maintenance

`third_party/whisper.cpp` is pinned to commit `30c5194c9691e4e9a98b3dea9f19727397d3f46e` (`v1.8.3-198-g30c5194c`) and should be maintained through the fork at `organic-programming/whisper.cpp`.

GitHub currently resolves the old `ggerganov/whisper.cpp` path to `ggml-org/whisper.cpp`. Use `ggml-org/whisper.cpp` for new upstream remotes.

## Branch policy

- `upstream`: read-only mirror of the canonical upstream `master`. Never commit to it directly.
- `dev`: default branch for our fork. Carry local patches here on top of a pinned upstream tag or commit.
- `master`: stable releases promoted from `dev` after validation.

## Local maintainer setup

```bash
cd third_party/whisper.cpp
git remote set-url origin git@github.com:organic-programming/whisper.cpp.git
git remote add upstream https://github.com/ggml-org/whisper.cpp.git
git fetch origin
git fetch upstream --tags
git checkout dev
```

If `upstream` already exists, update its URL instead:

```bash
git remote set-url upstream https://github.com/ggml-org/whisper.cpp.git
```

## Update cycle

1. Refresh the mirror branch:

   ```bash
   git fetch upstream --tags
   git checkout upstream
   git merge --ff-only upstream/master
   git push origin upstream
   ```

2. Update `dev` from an upstream release tag when possible:

   ```bash
   git checkout dev
   git merge v1.x.x
   ```

   If a release tag is not appropriate, merge the explicit upstream commit you intend to pin instead.

3. Test `wisupaa-whisper` against the updated submodule.
4. Push `dev` and review the result.
5. Merge `dev` into `master` on the fork when the update is stable.
