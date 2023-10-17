# DATS_6101 - Data Brew: Brewing Success with Starbucks Customer Insights
# Description of the Dataset: 
The Starbucks Customer Dataset comprises simulated interactions of 17,000 customers, offering insights into user engagement, the effectiveness of offers, and transaction behaviors on the Starbucks mobile app, encompassing diverse offers, distribution channels, and customer demographic data.
The data is contained in three files:
- portfolio.csv - data about offers sent to customers (10 offers x 6 columns)

- profile.csv - demographic data of customers (17,000 customers x 5 columns)

- transcript.csv - customer response to offers and transactions made (306,648 events x 4 columns)

Here is the schema and explanation of each variable in the files:

## profile.json
- age (int) - age of the customer
- became_member_on (int) - date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income
## portfolio.json
- id (string) - offer id
- offer_type (string) - type of offer ie BOGO, discount, informational
- difficulty (int) - minimum required spend to complete an offer
- reward (int) - reward given for completing an offer
- duration (int) - time for offer to be open, in days
- channels (list of strings)
## transcript.json
- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
- person (str) - customer id
- time (int) - time in hours since start of test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record
# SMART Questions: 
1. Does the combination of multiple communication channels significantly affect customer responsiveness, spending, and behavior within the dataset?
2. Is there a discernible correlation in the dataset between customers who receive 'BOGO' and 'discount' offers and their purchasing likelihood, and can we quantify the spending difference over one month?
3. How can Starbucks create personalized offers based on past purchases and explore the connection between demographic factors (age, gender, income) and offer response patterns?
4. In what ways can Starbucks leverage regional data to tailor marketing approaches and product offerings to meet the unique preferences of customers in different geographic locations?
5. How can Starbucks use regional data to customize marketing strategies and product offerings for diverse customer preferences across geographic areas?

