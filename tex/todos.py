with open('praca.tex', 'r') as f:
    f =f.read()
    fix_pat = "FIXME"
    todo_pat = "TODO"
    Ntodo = f.count(todo_pat)
    Nfix = f.count(fix_pat)
    print "Number of todos:", Ntodo 
    print "Number of fixes:", Nfix
    print "Overall: ", Ntodo + Nfix

