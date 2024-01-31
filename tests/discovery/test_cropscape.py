import act


def test_croptype():
    year = 2018
    lat = 37.15
    lon = -98.362
    # Try for when the cropscape API is not working
    try:
        crop = act.discovery.cropscape.get_crop_type(lat, lon, year)
        crop2 = act.discovery.cropscape.get_crop_type(lat, lon)
    except Exception:
        return

    # print(crop, crop2)
    if crop is not None:
        assert crop == 'Dbl Crop WinWht/Sorghum'
    if crop2 is not None:
        # assert crop2 == 'Sorghum'
        assert crop2 in ['Soybeans', 'Winter Wheat']
