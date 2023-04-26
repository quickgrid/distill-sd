from setuptools import setup, find_packages

setup(
    name='distillsd',
    author='Asif Ahmed',
    description='Distill latent diffusion model, stable diffusion text to image, image to image etc.',
    version='0.0.4',
    url='https://github.com/quickgrid/distill-sd',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'omegaconf',
        'safetensors',
        'torch',
        'torchvision',
        'pillow',
        'tqdm',
        'einops',
        'lightning',
        'xformers',
        'transformers'
    ]
)
