def build_org(array):
    org = {}

    for employees in array:
        employee_list = employees.split(' ')
        boss = employee_list[0]
        reports = employee_list[1:]
        # add the boss and their reports in org dict
        if boss not in org:
            org[boss] = []
        org[boss].extend(reports)
    
    return org

def print_org(org, boss, indent = 0):
    print("." * indent + boss)
    if boss in org:
        for report in org[boss]:
            print_org(org, report, indent=indent+5)
    return

def print_skip_level(org, boss, indent = 0):
    print("." * indent + boss)
    if boss in org:
        for report in org[boss]:
            if report in org:
                for skiplevel in org[report]:
                    print("." * (indent+5) + skiplevel)
    return   



def main():
    array = [
        "A B C",
        "B D",
        "D E",
        "F G",
        
    ]

    org = build_org(array)
    # print(org)
    # find the main boss - could be one person or many people who dont have any bosses above them
    # creating set of all employees with a boss
    employees = {report for report_list in org.values() for report in report_list}
    main_bosses = []
    for boss in org:
        if boss not in employees:
            main_bosses.append(boss)
    # print("Bosses: ", main_bosses)

    # print the organizational structure
    for boss in main_bosses:
        print_org(org, boss)


    # print skip level reports
    print("\nSkip level reports")
    boss_for_skip_levels = ['B', 'C']
    for boss in boss_for_skip_levels:
        print_skip_level(org, boss)

if __name__=="__main__":
    main()