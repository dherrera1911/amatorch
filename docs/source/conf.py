# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "amatorch"
copyright = "2024, Daniel Herrera-Esposito"
author = "Daniel Herrera-Esposito"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

templates_path = ["_templates"]
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# max time (in secs) per notebook cell. here, we disable this
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "home_page_in_toc": True,
    "github_url": "https://github.com/dherrera1911/amatorch",
    "repository_url": "https://github.com/dherrera1911/amatorch",
    "logo": {
        "alt_text": "Home",
        "image_light": "_static/amatorch.svg",
        "image_dark": "_static/amatorch_darkmode.svg",
    },
    "use_download_button": True,
    "use_repository_button": True,
}
