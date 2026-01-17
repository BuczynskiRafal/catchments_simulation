from django.contrib import admin

from .models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "bio_preview")
    search_fields = ("user__username", "user__email", "bio")
    list_filter = ("user__is_active",)
    raw_id_fields = ("user",)

    def bio_preview(self, obj):
        """Return first 50 characters of bio."""
        return obj.bio[:50] + "..." if len(obj.bio) > 50 else obj.bio

    bio_preview.short_description = "Bio Preview"
