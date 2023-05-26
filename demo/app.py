from src.gui import WebUI


def main():
    print("Launching demo...")

    # cwd = "/Users/andreped/workspace/livermask/"  # local testing -> macOS
    cwd = "/home/user/app/"  # production -> docker

    model_name = "model.h5"  # assumed to lie in `cwd` directory
    class_name = "parenchyma"

    # initialize and run app
    app = WebUI(model_name=model_name, class_name=class_name, cwd=cwd)
    app.run()


if __name__ == "__main__":
    main()
