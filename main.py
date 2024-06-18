from RV.RVHMM import reconocer_voz
from RV.conversor import convertidor
#from grabado import grabador


"""
    Si lo que quieres es grabar nuevos ejemplos
    comenta :
        import conversor.convertidor
        texto = reconocer_voz
        print(f"Se reconocio: {texto}")
        
    si lo que quieres es reconocer
    comenta : 
        from grabado import grabador
"""


def main():
    texto = reconocer_voz()
    print(f"Se reconocío : {texto}")
    print("grabacion finalizada")


if __name__ == "__main__":
    main()
