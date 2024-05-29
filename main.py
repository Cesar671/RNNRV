from RV.RV import reconocer_voz


def main():
    texto = reconocer_voz()
    print(f"se reconocío : {texto}")


if __name__ == "__main__":
    main()
