"""Run once in your local VS Code terminal to authorize Google GSC."""

from gsc_api import authorize, list_gsc_sites


def main():
    authorize()
    sites = list_gsc_sites()

    print("\nGoogle Search Console connection successful.")
    print(f"Accessible GSC properties: {len(sites)}\n")
    for site in sites:
        print(f"{site['siteUrl']}  ({site['permissionLevel']})")

    if not sites:
        print("No properties returned. Confirm this Google account has GSC access.")


if __name__ == "__main__":
    main()
