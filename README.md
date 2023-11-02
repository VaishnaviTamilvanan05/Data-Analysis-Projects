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
1. How do different promotional events (received, viewed, completed) affect customer retention and repeat business with Starbucks over time?             
2. What is the profile of the customer base, including their gender distribution, age distribution, and income range, and how do these factors relate to each other?
3. What are the most popular offers and the influential factors driving the View rates of the Offers?
4. Which age groups are more likely to accept offers than others? Do these preferences also differ based on gender?


