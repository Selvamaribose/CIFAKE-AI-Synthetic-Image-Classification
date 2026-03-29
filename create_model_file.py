from model_utils import MODEL_PATH, build_classifier_model


def main():
    model = build_classifier_model()
    model.save(MODEL_PATH)
    print(f"Saved model file to {MODEL_PATH}")


if __name__ == "__main__":
    main()
