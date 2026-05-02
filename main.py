import pandas as pd

def main():
    # there i have some problem with UTF-8
    df = pd.read_csv( "Sample - Superstore.csv", encoding='ISO-8859-1' )
    print( df.head( 10 ) )

    print( "Shape", df.shape )

if __name__ == "__main__":
    main()
