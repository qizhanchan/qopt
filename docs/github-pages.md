# GitHub Pages deployment

The site is built from `docs/` with MkDocs Material. The workflow in
`.github/workflows/pages.yml` builds and checks it on pull requests, then
deploys after a push to `main` or a manual run on `main`.

## Enable Pages

1. In the GitHub repository, open **Settings → Pages**.
2. Select **GitHub Actions** as the build and deployment source.
3. Merge or push the workflow to `main`. The published URL will be
   `https://qizhanchan.github.io/qopt/`.

The deploy job uses the `github-pages` environment and requires `pages: write`
and `id-token: write`. If the repository remains private, check that the
account plan supports Pages for private repositories. A Pages site can still
be public even when its source repository is private.

## Preview locally

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r docs/requirements.txt
.venv/bin/mkdocs build --strict
.venv/bin/mkdocs serve
```

The generated `site/` directory is ignored by Git. The workflow uploads it
as a Pages artifact; it does not write generated files to a branch.

See [GitHub's Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
for the required repository setting and deployment permissions.
