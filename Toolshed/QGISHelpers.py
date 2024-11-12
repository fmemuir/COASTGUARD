# Get the active layer
layer = iface.activeLayer()

# Check if there's a selected feature
selfeats = layer.selectedFeatures()
if not selfeats:
    print("Please select a feature first.")
else:
    # Start editing
    layer.startEditing()
    
    # Assuming one feature is selected
    for selfeat in selfeats:
        dateval = selfeat["dates"]  
        timeval = selfeat["times"]

        # Iterate over all features and set pos_error for matching dates
        for f in layer.getFeatures():
            if f["dates"] == dateval and f["times"] == timeval:  # Match date and time
                layer.changeAttributeValue(f.id(), layer.fields().indexFromName("pos_error"), 1)

    # Commit changes
    layer.commitChanges()
