"""Keep CAD stop filtering outside the subtree replicated by PhysX."""


def relocate_collision_groups(stage, env_paths):
    from pxr import Sdf, UsdPhysics

    source_robot = env_paths[0] + '/Robot'
    source_path = source_robot + '/collision_filters'
    source = stage.GetPrimAtPath(source_path)
    if not source or not source.IsActive():
        return
    destination = '/World/mine_collision_filters'
    groups = [p for p in source.GetChildren() if p.IsA(UsdPhysics.CollisionGroup)]
    if not groups:
        return
    # CollisionGroup objects cannot be replicated by the PhysX cloner. Put
    # three shared groups on the scene and explicitly include every clone's
    # shapes; ground stays outside them, preserving ground contacts.
    for prim in groups:
        group = UsdPhysics.CollisionGroup.Define(stage, destination + '/' + prim.GetName())
        original = UsdPhysics.CollisionGroup(prim)
        for attribute in prim.GetAttributes():
            if attribute.HasAuthoredValueOpinion():
                group.GetPrim().CreateAttribute(attribute.GetName(), attribute.GetTypeName(),
                                               custom=attribute.IsCustom()).Set(attribute.Get())
        filtered = original.GetFilteredGroupsRel().GetTargets()
        group.CreateFilteredGroupsRel().SetTargets([
            Sdf.Path(destination + '/' + path.name) for path in filtered
        ])
        includes = original.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()
        suffixes = []
        for path in includes:
            if not str(path).startswith(source_robot + '/'):
                raise ValueError('Stop collision group includes a shape outside the source robot')
            suffixes.append(str(path)[len(source_robot):])
        group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([
            Sdf.Path(env_path + '/Robot' + suffix)
            for env_path in env_paths for suffix in suffixes
        ])
    source.SetActive(False)
