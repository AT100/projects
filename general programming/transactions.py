import json


class BillingStatus:
    def __init__(self, user_id = None, ad_delivery_pennies = 0, payment_pennies = 0, transactions = {}):
        self.user_id = user_id
        self.ad_delivery_pennies = ad_delivery_pennies
        self.payment_pennies = payment_pennies
        self.undone_transaction = {}
        self.transactions = transactions

    def transactionProcessing(self, input_transaction):
        self.transactions[input_transaction.get('transaction_timestamp', 0)] = input_transaction

        self.ad_delivery_pennies += input_transaction.get("ad_delivery_pennies", 0)
        self.payment_pennies += input_transaction.get("payment_pennies", 0)
        self.overwriteTransaction(input_transaction)

        self.undo(input_transaction)
        self.redo(input_transaction)
       

    def overwriteTransaction(self, transaction):
        if 'overwrite' in transaction and transaction['overwrite']:
            self.ad_delivery_pennies = transaction['ad_delivery_pennies'] if 'ad_delivery_pennies' in transaction else self.ad_delivery_pennies
            self.payment_pennies = transaction['payment_pennies'] if 'payment_pennies' in transaction else self.payment_pennies
        return

    def undo(self, transaction):
        if 'undo_last' in transaction and transaction['undo_last']:
            prev_timestamp = transaction['transaction_timestamp'] - 1
            while prev_timestamp in self.transactions:
                prev_transaction = self.transactions[prev_timestamp]
                #check if its a regular transaction or not
                if 'undo_last' in prev_transaction or 'redo_last' in prev_transaction or ('overwrite' in prev_transaction and 'user_id' in prev_transaction):
                    print(f"Non regular transaction at {prev_timestamp}")
                    prev_timestamp -= 1
                    print("Checking {prev_timestamp}")
                else:
                    print(f"self.ad_delivery_pennies {self.ad_delivery_pennies}")
                    self.ad_delivery_pennies -=  prev_transaction.get('ad_delivery_pennies', 0)
                    self.payment_pennies -= prev_transaction.get('payment_pennies', 0)
                    print(f"self.ad_delivery_pennies {self.ad_delivery_pennies}")
                    self.undone_transaction = prev_transaction
                    break


    def redo(self, transaction):
        if 'redo_last' in transaction and transaction['redo_last']:
            self.ad_delivery_pennies +=  self.undone_transaction.get('ad_delivery_pennies', 0)
            self.payment_pennies +=  self.undone_transaction.get('payment_pennies', 0)

    def __rep__(self):
        return f"{self.user_id} : BillingStatus('ad_delivery_pennies': {self.ad_delivery_pennies}, 'payment_pennies': {self.payment_pennies})"

def main():
    with open('trans.json') as f:
        input = json.load(f)

    user_accounts = {}

    for id, transaction in input.items():
        user_id = transaction['user_id']
        if user_id not in user_accounts:
            user_accounts[user_id] = BillingStatus(user_id=user_id)
        
        #user_accounts[user_id].transactions[transaction["transaction_timestamp"]] = transaction
        user_accounts[user_id].transactionProcessing(transaction)

    # #create some users
    # for user in user_accounts:
    #     user.transactionProcessing()

if __name__ == "__main__":
    main()
