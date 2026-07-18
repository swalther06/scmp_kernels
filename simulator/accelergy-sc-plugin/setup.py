from setuptools import setup

PLUG_IN_NAME = "accelergy-sc-plugin"

setup(
    name=f"{PLUG_IN_NAME}",
    version="0.1",
    description="Accelergy energy/area estimator for a unary-stochastic-computing MAC.",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
    keywords="accelerator hardware energy estimation stochastic computing timeloop accelergy",
    license="MIT",
    install_requires=["accelergy>=0.4"],
    python_requires=">=3.8",
    data_files=[
        (
            f"share/accelergy/estimation_plug_ins/{PLUG_IN_NAME}",
            [
                "sc_mac.py",
                "rng_bank.py",
                "sc_mac.estimator.yaml",
            ],
        ),
    ],
    include_package_data=True,
    entry_points={},
    zip_safe=False,
)
