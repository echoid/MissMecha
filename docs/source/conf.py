# Configuration file for the Sphinx documentation builder.

import os
import sys
from sphinx_gallery.sorting import FileNameSortKey

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../missmecha"))

# -- Project information -----------------------------------------------------

project = "MissMecha"
copyright = "2025, Youran Zhou"
author = "Youran Zhou"
release = "0.0.1"


# 项目名称：决定左上角显示什么
project = "MissMecha"

# 文档标题：影响浏览器标题和顶部文字
html_title = "MissMecha"

# 可选：缩短左上角导航栏的名字（不影响浏览器标签）
html_short_title = "MissMecha"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
     'nbsphinx',
    'sphinx.ext.mathjax',

]


templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
#html_theme = "pydata_sphinx_theme"
html_theme = "pydata_sphinx_theme"

#extensions += ["sphinx_gallery.gen_gallery"]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",       # your example script location
    "gallery_dirs": "auto_examples",      # output folder with HTML + downloads
    "filename_pattern": r"demo.*\.py",   # files that are processed
    "download_all_examples": False,       # optional, disables ZIP
}

autodoc_duplicate_warn = False



html_theme_options = {
    "navbar_align": "left",
    "navbar_end": ["navbar-icon-links"],  # 去掉 theme-switcher
    "navigation_with_keys": True,
        "collapse_navigation": False,  # <<< 关键配置！
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/echoid/MissMecha",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/missmecha-py/",
            "icon": "fas fa-box",
            "type": "fontawesome",
        },
    ],
}



html_static_path = ['../_static']
html_css_files = ['css/custom.css']  # ✅ 这是相对于 _static 目录的路径
