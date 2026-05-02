import pandas as pd

def main():
    # there i have some problem with UTF-8
    df = pd.read_csv( "data/Sample - Superstore.csv", encoding='ISO-8859-1' )
    
    ### Assignment 2
    ## ~~ Task № 1
    # Show 10 rows of data
    # print( df.head( 10 ) )

    # Show shape inf
    # print( "\nShape:", df.shape )
    
    # Columns type info
    # print( df.info() )

    # Show statist inf
    # print( "Statistic:\n", df.describe() )

    # Show empty cell on data
    
    # Empty procent
    # print(  df.isnull().sum() / len(df) * 100 )
    # print( type( df.isnull().sum() ) )

    """        
    print( "Was:" )
    print( df  )

    df[ 'Sales' ].fillna(
        df[ 'Sales' ].median(),
        inplace = True
    )

    # print( median_sales )

    df[ "Ship Mode" ].fillna(
        df[ 'Ship Mode' ].mode()[ 0 ],
        inplace = True
    )
    
    print( "\nAfter:" )
    print( df  )
    """


    ## ~~ Task # 3: Handle Missing Values  
    
    print( "\n\n" + "~" * 50 ) 
    print( "### Task 3: Handle Missing Values" )  
    print( "~" * 50 )

    non_counts =  df.isnull().sum()
   
    # Find empty columns
    for column_n, empty_v, in non_counts.items():
        print( column_n, empty_v )
        print( df[ column_n ].dtype )

        # Get with empty
        if empty_v > 0:
            print( column_n, "<- Empty: ", empty_v  )
            print( "=" * 30 )            
            
            # For the numerical column
            if str( df[ column_n ].dtype ) in [ 'int64', 'float64' ]:
                df[ column_n ] = df[ column_n ].fillna( df[ column_n ].median() )
                print( "\n### Median ###\n" )
            
            # For the categorical columns
            else:
                df[ column_n ].fillna( df[ column_n ].mode()[ 0 ], inplace = True )
                print( "\n### Mode ###\n" )
       
        # Non Empty
        else:
            print( column_n, "<- No Empty: ", empty_v  )
            print( "=" * 30 )            
   
    ## ~~ Task № 4: Remove Duplicates 
    
    print( "\n\n" + "~" * 50 )
    print( "### Task 4: Remove Duplicates" ) 
    print( "~" * 50 )

    if df.duplicated().sum() > 0:
        print( "Duplicate rows <- ", df.duplicated().sum() )
        df = df.drop_duplicates()
        
        print( "Shape after removing duplicates: ", df.shape )

    else:
        print( "No Duplicate rows <- ", df.duplicated().sum() )

    ## ~~ Task № 5: Convert Data Types
    
    print( "\n\n" + "~" * 50 )
    print( "### Task 5: Convert Data Types" ) 
    print( "~" * 50 )


    df[ 'Order Date' ] = pd.to_datetime( df[ 'Order Date' ] )
    df[ 'Ship Date' ]  = pd.to_datetime( df[ 'Ship Date' ] ) 

    # Check types
    print( df.dtypes )

    # Check data
    print( df[ ['Order Date', 'Ship Date'] ].head() )


    ## ~~ Task № 6: Create Derived Features
    
    print( "\n\n" + "~" * 50 )
    print( "### Task 6: Create Derived Features" ) 
    print( "~" * 50 )

    # Total spending customer
    total_spend = df.groupby( 'Customer ID' )[ 'Sales' ].sum().reset_index()
    total_spend.columns = [ 'Customer ID', 'Total_Spending' ]

    # Order frequency
    order_freq = df.groupby( 'Customer ID' )[ 'Order ID' ].nunique().reset_index()
    order_freq.columns = [ 'Customer ID', 'Order_Frequency' ]

    # Average order vl
    agv_order = df.groupby( 'Customer ID' )[ 'Order ID' ].mean().reset_index()
    agv_order.columns = [ 'Customer ID', 'Avg_Order_Value' ]

    # Merge
    customer_merge = total_spend.merge( order_freq, on = 'Customer ID' ).merge( agv_order, on = 'Customer ID' )
    print( customer_merge.head( 10 ) )

    df.to_csv( 'data/cleaned_store_data.csv', index = False )
    customer_merge.to_csv( 'data/customer_merge.csv', index = False )

if __name__ == "__main__":
    main()
