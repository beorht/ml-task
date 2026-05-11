## ~~~ Section 1 ~~~
"""
import pandas as pd

## Example dataset
df = pd.DataFrame({
    "name": [ "Behruz", "Bekzod", "Jasur", "Otabek", "Samandar" ],
    "age": [ 24, 28, 31, 89, 12 ],
    "city": [ "Namangan", "Samarqand", "Bukhara", "Namangan", "Jizzax" ]
})


## New one for add
new_rows = pd.DataFrame({
    "name": [ "Rayxon", "Gulxayo", "Rustam" ],
    "age": [ 19, 5, 19 ],
    "city": [ "Tashkent", "Andijon", "Buxoro" ]
})

## Merge two dataset
df = pd.concat( [ df, new_rows ], ignore_index = True )

print( df.head(10) )
"""

## ~~~ Section 2 ~~~
"""
import pandas as pd

# Getting online dataset
df = pd.read_csv( "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv" )

print( df.head(20) )      # show 20 rows of dataset
print( df.shape )         # show inf about rows and colums
print( df.info() )        # show inf about type of dataset
print( df.describe() )    # statistic (count, mean, std, min, 25%, 50%, 70%, max)
print( df.isna().sum() )  # show missing values from each columns 
"""

## ~~~ Section 3 ~~~
import pandas as pd

# Getting online dataset
df = pd.read_csv( "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv" )

print( df.columns )

### ~~ 3.1 ~~
# Select Columns for Series
ser = df[ "Age" ]

sub_df = df[ [ "Name", "Survived" ] ]

print( ser, "\n" )
print( sub_df, "\n" )

row_label = df.loc[ 0 ] # first row label
print( row_label, "\n" )

first_row = df.iloc[ 0 ] # first row position
print( first_row, "\n" )

### ~~ 3.2 ~~
# filtered
filtered1 = df[ ( df[ "Age" ] > 30 ) & ( df[ "Sex" ] == "female" ) ]
filtered2 = df[ ( df[ "Pclass" ] == 1 ) | ( df[ "Survived" ] == 1 ) ]

print( filtered1, "\n" )
print( filtered2, "\n" )

### ~~ 3.3 ~~
# Getting Series
avg_age_by_class = df.groupby( "Pclass" )[ "Age" ].mean()
print( avg_age_by_class, "\n" )

# Getting DataFrame
summary = df.groupby( "Pclass" ).agg({
    "Age": "mean",
    "PassengerId": "count"
})
print( summary, "\n" )

### ~~ 3.4 ~~~
# New DataFrame
lookup = pd.DataFrame({
    "Pclass": [ 1, 2, 3 ],
    "class_name": [ "First", "Second", "Third" ],
    "ticket_price_range": [ "High", "Medium", "Low" ]
})

# Merge to df old DataFrame
merged = pd.merge( df, lookup, on = "Pclass", how = "left" )
print( merged )
