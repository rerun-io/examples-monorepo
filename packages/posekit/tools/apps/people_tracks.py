import tyro

from posekit.apis.people_tracks import PeopleTracksConfig, main

if __name__ == "__main__":
    main(tyro.cli(PeopleTracksConfig))
