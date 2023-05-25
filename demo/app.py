from src.gui import WebUI


def main():
    print("Launching demo...")

    model_name = "/home/user/app/model.h5"  # "/Users/andreped/workspace/livermask/model.h5"
    class_name = "parenchyma"

    # initialize and run app
    app = WebUI(model_name=model_name, class_name=class_name)
    app.run()


if __name__ == "__main__":
    main()
