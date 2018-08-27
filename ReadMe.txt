First need to convert the field `airline sentiment" to numeric so that we can find the average sentiment for each airline. We will use a conversion scheme as follows:
Sentiment Numeric Value
Positive 5.0
Neutral 2.5
Negative 1.0
Using the above scheme, you will calculate the average rating for each airline, and then figure out which are the best and worst in terms of average ratings. If there is a tie, you can choose any one at random. You will then do topic modeling on these two airlines using the \text" field, with the stop words removed.