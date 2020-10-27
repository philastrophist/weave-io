from weaveio.basequery.common import NotYetImplementedError, FrozenQuery


class BranchLogic(FrozenQuery):
    def __init__(self, parent):
        super(BranchLogic, self).__init__(parent=parent)
        self.branches = []

    def simplify(self):
        pass

