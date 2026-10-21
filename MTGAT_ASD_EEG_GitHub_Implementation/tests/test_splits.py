from mtgat.splits import participant_split

def test_no_participant_overlap():
    ids=[f"sub-{i//3}" for i in range(60)]
    tr,va,te=participant_split(ids,seed=7)
    groups=[set(ids[i] for i in x) for x in (tr,va,te)]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
