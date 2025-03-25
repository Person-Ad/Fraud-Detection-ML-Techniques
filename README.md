# IEEE-CIS Fraud Detection
[Dataset Link](https://www.kaggle.com/competitions/ieee-fraud-detection/data)


## Evaluation
- main one AUC (area under the ROC curve) between the predicted probability and the observed target

## Output
probability of this transaction `isFraud` or not

## Data
The data is broken into two files `identity` and `transaction`, which are joined by `TransactionID`. Not all transactions have corresponding identity information.
### Columns - Identity
    TransactionID
    id_01 - id_38
    DeviceType: eg(mobile, desktop, etc..)
    DeviceInfo: eg(Windows, Samsung, etc..)
### Columns - Transaction
    TransactionID
    isFraud
    TransactionDT
    TransactionAmt
    ProductCD
    card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
    addr1 - addr2 	
    dist1 - dist2
    P_emaildomain
    R_emaildomain
    C1 - C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked
    D1-D15: timedelta, such as days between previous transaction, etc. 
    M1-M9: match, such as names on card and address, etc.
    Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

### Categorical Features - Transaction
    ProductCD:      Product code, the product for each transaction 
    emaildomain
    card1 - card6:  payment card information, such as card type, card category, issue bank, country, etc. 
    addr1, addr2
    P_emaildomain:  purchaser email domain
    R_emaildomain:  recipient email domain
    M1 - M9:        match, such as names on card and address, etc.
### Categorical Features - Identity
    DeviceType
    DeviceInfo
    id_12 - id_38: 
