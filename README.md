# DATS_6101 - Data Brew: Brewing Success with Starbucks Customer Insights
# Proposal: 
The Starbucks Customer Dataset simulates interactions and transactions to optimize promotional strategies. It includes diverse offers sent through different channels to 17,000 customers with demographic details. Events capture customer responses and transactions, enabling analysis of offer effectiveness and customer behavior.

Our project aims to determine optimal promotion channels, tailor offers based on customer attributes and actions, and elevate the overall efficiency and effectiveness of Starbucks' marketing efforts. This initiative seeks to maximize customer engagement, elevate customer satisfaction, and drive the success of Starbucks' marketing strategies.

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
2. Are "BOGO" and "discount" offers correlated with the possibility that a customer would make a purchase?
3. How can Starbucks personalize offers using past purchases and study demographic factors' impact on offer response?
4. What's the average time between receiving an offer and viewing it, and does this time gap affect conversion rates?


