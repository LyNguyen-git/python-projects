
def contact_book():
    contacts = {1:['Tim Nielsen','12345678'],2:['Foo Bar','9876544321']}
    return contacts

def main():
    contacts = contact_book()
    contact = []
    idx = 0
    stop = False
    while not stop:
        get_input = False
        while not get_input:
            try:
                idx = int(input('Which contact do you want to modify: '))
                try:
                    contact = contacts[idx]
                    get_input = True
                except:
                    print('This contact not exist.')
            except ValueError as e:
                print('Cannot get valid input. {}'.format(e))      

        print(f'Current values: Name {contact[0]}, Phone number {contact[1]}')     
        new_name = input('New name: ')
        if new_name == '':
            print('Name not changed')
        else:
            contact[0] = new_name
            print(f'Name has changed to {new_name}')

        new_phone_number = input('New phone number: ')
        if new_phone_number == '':
            print('Phone number not changed')
        else:
            contact[1] = new_phone_number
            print(f'Changed phone number to {new_phone_number}')
        
        command = input('\'x\' to quit, Enter to continue :')
        print('\n')
        if command == 'x':
            stop = True

if __name__=="__main__":
    main()