import setuptools
from setuptools.command.develop import develop
import subprocess

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

with open("requirements_local.txt", "r") as f:
    requirements_local = f.read().splitlines()

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="HiSim",
    version="0.1.1",
    author="xymou",
    author_email="xymou20@fudan.edu.cn",
    description="A hybrid social media simulation framework based on AgentVerse and mesa",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xymou/social_simulation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    # install_requires=[
    #     "PyYAML",
    #     "fastapi",
    #     "uvicorn",
    #     "py3langid",
    #     "iso-639",
    #     "openai",
    #     "opencv-python",
    #     "gradio",
    #     "httpx[socks]",
    #     "astunparse",
    #     "langchain",
    # ],
    install_requires=requirements,
    extras_require={
        'local': requirements_local
    },
    include_package_data = True,
    entry_points={
        "console_scripts": [
            "agentverse-microtest = agentverse_command.main_microtest_cli:cli_main",
            "agentverse-simulation = agentverse_command.main_simulation_cli:cli_main",
        ],
    },
)
