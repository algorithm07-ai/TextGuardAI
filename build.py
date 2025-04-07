import os
import subprocess
import shutil

def clean_build():
    """Clean build directories."""
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

def build_package():
    """Build the package."""
    subprocess.run(['python', 'setup.py', 'sdist', 'bdist_wheel'], check=True)

def main():
    """Main build process."""
    print("Cleaning build directories...")
    clean_build()
    
    print("Building package...")
    build_package()
    
    print("Build complete! Distribution files are in the 'dist' directory.")

if __name__ == '__main__':
    main() 